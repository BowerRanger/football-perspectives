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
from src.utils.pose_fusion import (
    robust_translation_fuse,
    so3_chordal_mean,
    so3_geodesic_distance,
)

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
        # Project each contribution onto the reference timeline.
        on_ref: list[tuple[str, SmplWorldTrack, np.ndarray]] = []
        for shot_id, track in contribs:
            offset = sync_map.offset_for(shot_id) if shot_id else 0
            ref_frames = np.asarray(track.frames, dtype=np.int64) - offset
            on_ref.append((shot_id, track, ref_frames))

        union = np.unique(np.concatenate([rf for _, _, rf in on_ref]))
        n = len(union)

        fused_root_t = np.zeros((n, 3))
        fused_root_R = np.tile(np.eye(3), (n, 1, 1))
        fused_thetas = np.zeros((n, 24, 3))
        fused_conf = np.zeros(n)
        view_count = np.zeros(n, dtype=np.int32)

        # Per-shot frame → row index lookup, for O(1) access at each ref frame.
        lookups: list[tuple[str, SmplWorldTrack, dict[int, int]]] = [
            (sid, tr, {int(f): i for i, f in enumerate(rf)})
            for (sid, tr, rf) in on_ref
        ]

        diag_frames: list[FrameDiagnostic] = []

        for i, ref in enumerate(union):
            views: list[
                tuple[str, np.ndarray, np.ndarray, np.ndarray, float]
            ] = []
            for sid, tr, idx in lookups:
                local_i = idx.get(int(ref))
                if local_i is None:
                    continue
                conf = float(tr.confidence[local_i])
                if conf <= 0.0:
                    continue
                views.append(
                    (
                        sid,
                        np.asarray(tr.root_t[local_i], dtype=np.float64),
                        np.asarray(tr.root_R[local_i], dtype=np.float64),
                        np.asarray(tr.thetas[local_i], dtype=np.float64),
                        conf,
                    )
                )

            if not views:
                continue

            sids = tuple(v[0] for v in views)
            if len(views) == 1:
                _, t, R, thetas, conf = views[0]
                fused_root_t[i] = t
                fused_root_R[i] = R
                fused_thetas[i] = thetas
                fused_conf[i] = conf
                view_count[i] = 1
                diag_frames.append(
                    FrameDiagnostic(
                        frame=int(ref),
                        contributing_shots=sids,
                        dropped_shots=(),
                        pos_disagreement_m=0.0,
                        rot_disagreement_rad=0.0,
                        low_coverage=True,
                        high_disagreement=False,
                    )
                )
                continue

            # Multi-view path with MAD-based outlier rejection on positions.
            weights = np.array([v[4] for v in views])
            ts = np.stack([v[1] for v in views])
            Rs = np.stack([v[2] for v in views])
            thetass = np.stack([v[3] for v in views])

            fused_t, kept = robust_translation_fuse(
                ts, weights, k_sigma=float(cfg.get("outlier_k_sigma", 3.0))
            )
            kept_sids = tuple(v[0] for v, k in zip(views, kept) if k)
            dropped_sids = tuple(v[0] for v, k in zip(views, kept) if not k)
            kept_idx = np.where(kept)[0]
            kept_weights = weights[kept_idx]
            kept_Rs = Rs[kept_idx]
            kept_thetas = thetass[kept_idx]
            fused_R = so3_chordal_mean(kept_Rs, kept_weights)

            joint_R_per_view = np.stack(
                [
                    _axis_angle_to_so3_batch(kept_thetas[v])
                    for v in range(len(kept_idx))
                ]
            )  # (Vk, 24, 3, 3)
            fused_joint_R = np.stack(
                [
                    so3_chordal_mean(joint_R_per_view[:, j], kept_weights)
                    for j in range(24)
                ]
            )
            fused_theta = np.stack(
                [_so3_to_axis_angle(fused_joint_R[j]) for j in range(24)]
            )

            fused_root_t[i] = fused_t
            fused_root_R[i] = fused_R
            fused_thetas[i] = fused_theta
            fused_conf[i] = float(kept_weights.sum())
            view_count[i] = int(len(kept_idx))

            kept_ts = ts[kept_idx]
            pos_dis = (
                float(np.max(np.linalg.norm(kept_ts - fused_t, axis=1)))
                if len(kept_ts) > 0
                else 0.0
            )
            rot_dis = (
                float(
                    max(
                        so3_geodesic_distance(kept_Rs[v], fused_R)
                        for v in range(len(kept_idx))
                    )
                )
                if len(kept_idx) > 0
                else 0.0
            )
            min_views = int(cfg.get("min_contributing_views", 1))
            diag_frames.append(
                FrameDiagnostic(
                    frame=int(ref),
                    contributing_shots=kept_sids,
                    dropped_shots=dropped_sids,
                    pos_disagreement_m=pos_dis,
                    rot_disagreement_rad=rot_dis,
                    low_coverage=(int(len(kept_idx)) < min_views),
                    high_disagreement=(
                        pos_dis > float(cfg.get("high_disagreement_pos_m", 0.5))
                        or rot_dis
                        > float(cfg.get("high_disagreement_rot_rad", 0.5))
                    ),
                )
            )

        # Drop frames where all views had zero confidence (view_count stayed 0).
        keep = view_count > 0
        union = union[keep]
        fused_root_t = fused_root_t[keep]
        fused_root_R = fused_root_R[keep]
        fused_thetas = fused_thetas[keep]
        fused_conf = fused_conf[keep]
        view_count = view_count[keep]

        # Beta fusion: weighted mean across all contributing tracks.
        beta_stack = np.stack([np.asarray(tr.betas) for _, tr in contribs])
        beta_weights = np.array(
            [float(np.asarray(tr.confidence).mean()) for _, tr in contribs]
        )
        if beta_weights.sum() > 0:
            fused_betas = (
                (beta_weights[:, None] * beta_stack).sum(axis=0) / beta_weights.sum()
            )
        else:
            fused_betas = beta_stack.mean(axis=0)

        if len(contribs) > 1:
            max_beta_dist = float(
                np.max(
                    [
                        np.linalg.norm(beta_stack[i] - fused_betas)
                        for i in range(len(contribs))
                    ]
                )
            )
            if max_beta_dist > float(cfg.get("beta_disagreement_warn", 0.3)):
                logger.warning(
                    "[refined_poses] %s beta disagreement %.3f exceeds %.3f",
                    player_id,
                    max_beta_dist,
                    cfg.get("beta_disagreement_warn", 0.3),
                )

        # Smoothing — wired in Task 7.
        contributing_shots = tuple(sorted({sid for sid, _ in contribs if sid}))

        refined = RefinedPose(
            player_id=player_id,
            frames=union,
            betas=fused_betas,
            thetas=fused_thetas,
            root_R=fused_root_R,
            root_t=fused_root_t,
            confidence=fused_conf,
            view_count=view_count,
            contributing_shots=contributing_shots,
        )
        diag = RefinedPoseDiagnostics(
            player_id=player_id,
            contributing_shots=contributing_shots,
            frames=tuple(diag_frames),
            summary={
                "total_frames": int(len(union)),
                "single_view_frames": int(
                    sum(1 for f in diag_frames if f.low_coverage)
                ),
                "high_disagreement_frames": int(
                    sum(1 for f in diag_frames if f.high_disagreement)
                ),
            },
        )
        return refined, diag


# ----------------------------------------------------------------------
# Rodrigues helpers — local to this module so the public utils stay clean.


def _axis_angle_to_so3(omega: np.ndarray) -> np.ndarray:
    """Rodrigues. omega is (3,). Returns (3, 3)."""
    theta = float(np.linalg.norm(omega))
    if theta < 1e-9:
        return np.eye(3)
    k = omega / theta
    K = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _axis_angle_to_so3_batch(thetas: np.ndarray) -> np.ndarray:
    """thetas: (24, 3) → (24, 3, 3)."""
    return np.stack([_axis_angle_to_so3(thetas[j]) for j in range(thetas.shape[0])])


def _so3_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Inverse of Rodrigues. Returns (3,)."""
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-9:
        return np.zeros(3)
    if abs(theta - np.pi) < 1e-6:
        diag = np.diag(R)
        i = int(np.argmax(diag))
        e = np.zeros(3)
        e[i] = 1.0
        v = (R[:, i] + e) / np.sqrt(max(2.0 * (1.0 + diag[i]), 1e-12))
        return theta * v
    sin_theta = np.sin(theta)
    omega = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return omega * theta / (2.0 * sin_theta)

"""HMR-in-pitch-frame stage.

Per-player monocular SMPL reconstruction expressed in pitch-world coords.

For each player track in ``output/tracks/PXXX_track.json``:
1. Run GVHMR over the track to obtain per-frame SMPL params in the camera
   frame (root rotation, pose, shape).
2. Median-aggregate the (per-frame-noisy) shape parameters.
3. Convert root rotation from camera frame to pitch frame via the calibrated
   camera extrinsic, then SLERP-smooth.
4. Savgol-smooth the per-joint axis-angle pose.
5. Compute per-frame translation by ankle-anchoring: project the 2D ankle
   midpoint to the pitch ground plane (z = 0.05 m) and back-solve the root
   translation that places the foot exactly there.
6. Ground-snap z when the avatar is roughly stationary.

Output: one ``SmplWorldTrack`` per player at
``output/hmr_world/{player_id}_smpl_world.npz``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.camera_track import CameraTrack
from src.schemas.smpl_world import SmplWorldTrack
from src.utils.foot_anchor import ankle_ray_to_pitch, anchor_translation
from src.utils.smpl_pitch_transform import smpl_root_in_pitch_frame
from src.utils.temporal_smoothing import (
    ground_snap_z,
    savgol_axis,
    slerp_window,
)

# Indices of left/right ankle in COCO 17 keypoints.
_COCO_LEFT_ANKLE = 15
_COCO_RIGHT_ANKLE = 16

# Ankle-confidence cutoff below which we mark a frame as low-confidence and
# do not anchor (matches the spec keypoint-confidence threshold).
_ANKLE_CONF_MIN = 0.3

# Foot offset relative to root, expressed in the pitch-world (z-up) frame
# *after* the SMPL→pitch static transform has been applied via root_R.
# See decision-log D9: the root frame here is pitch-world (z-up), so "foot
# below root" is along pitch -z, not the SMPL canonical -y.
_FOOT_IN_ROOT = np.array([0.0, 0.0, -0.95], dtype=float)

# Pitch ground-plane offset for the foot ray-cast (foot rests slightly above
# the turf surface so a vertical ray-from-camera doesn't grazing-intersect).
_FOOT_PLANE_Z = 0.05


class HmrWorldStage(BaseStage):
    name = "hmr_world"

    def is_complete(self) -> bool:
        out = self.output_dir / "hmr_world"
        return out.exists() and any(out.glob("*_smpl_world.npz"))

    def run(self) -> None:
        cfg = self.config.get("hmr_world", {})
        track_dir = self.output_dir / "tracks"
        camera_path = self.output_dir / "camera" / "camera_track.json"
        pose_dir = self.output_dir / "pose_2d"
        out_dir = self.output_dir / "hmr_world"
        out_dir.mkdir(parents=True, exist_ok=True)

        camera_track = CameraTrack.load(camera_path)
        per_frame_K = {f.frame: np.array(f.K, dtype=float) for f in camera_track.frames}
        per_frame_R = {f.frame: np.array(f.R, dtype=float) for f in camera_track.frames}
        t_world = np.array(camera_track.t_world, dtype=float)

        min_track_frames = int(cfg.get("min_track_frames", 10))
        savgol_window = int(cfg.get("theta_savgol_window", 11))
        savgol_order = int(cfg.get("theta_savgol_order", 2))
        slerp_w = int(cfg.get("root_slerp_window", 5))
        ground_snap_velocity = float(cfg.get("ground_snap_velocity", 0.1))

        for player_track_path in sorted(track_dir.glob("P*_track.json")):
            self._process_player(
                player_track_path=player_track_path,
                pose_dir=pose_dir,
                out_dir=out_dir,
                cfg=cfg,
                per_frame_K=per_frame_K,
                per_frame_R=per_frame_R,
                t_world=t_world,
                min_track_frames=min_track_frames,
                savgol_window=savgol_window,
                savgol_order=savgol_order,
                slerp_w=slerp_w,
                ground_snap_velocity=ground_snap_velocity,
            )

    def _process_player(
        self,
        *,
        player_track_path: Path,
        pose_dir: Path,
        out_dir: Path,
        cfg: dict,
        per_frame_K: dict[int, np.ndarray],
        per_frame_R: dict[int, np.ndarray],
        t_world: np.ndarray,
        min_track_frames: int,
        savgol_window: int,
        savgol_order: int,
        slerp_w: int,
        ground_snap_velocity: float,
    ) -> None:
        with player_track_path.open() as fh:
            track_data = json.load(fh)
        player_id = track_data["player_id"]
        track_frames: list[tuple[int, tuple[int, int, int, int]]] = [
            (int(f["frame"]), tuple(f["bbox"])) for f in track_data["frames"]
        ]
        if len(track_frames) < min_track_frames:
            return

        # 1. GVHMR per track (lazy import — heavy dependency).
        from src.utils.gvhmr_estimator import run_on_track

        shots = self.output_dir / "shots"
        video_candidates = sorted(shots.glob("*.mp4"))
        if not video_candidates:
            raise RuntimeError(
                f"no shot video found in {shots} — run prepare_shots first"
            )
        video_path = video_candidates[0]

        hmr_out = run_on_track(
            track_frames=track_frames,
            video_path=video_path,
            checkpoint=Path(cfg.get("checkpoint", "")),
            device=str(cfg.get("device", "auto")),
            batch_size=int(cfg.get("batch_size", 16)),
            max_sequence_length=int(cfg.get("max_sequence_length", 120)),
        )
        thetas = np.asarray(hmr_out["thetas"])             # (N, 24, 3)
        betas_all = np.asarray(hmr_out["betas"])           # (N, 10)
        root_R_cam = np.asarray(hmr_out["root_R_cam"])     # (N, 3, 3)
        joint_conf = np.asarray(hmr_out["joint_confidence"])  # (N, 24)

        # 2. Median shape across track.
        betas = np.median(betas_all, axis=0)

        # 3. Convert root rotation to pitch frame and SLERP-smooth.
        frame_indices = np.array([fi for fi, _ in track_frames])
        root_R_pitch = np.empty_like(root_R_cam)
        camera_present = np.zeros(len(frame_indices), dtype=bool)
        for i, fi in enumerate(frame_indices):
            R_t = per_frame_R.get(int(fi))
            if R_t is None:
                root_R_pitch[i] = np.eye(3)
                continue
            camera_present[i] = True
            root_R_pitch[i] = smpl_root_in_pitch_frame(root_R_cam[i], R_t)
        root_R_pitch = slerp_window(root_R_pitch, window=slerp_w)

        # 4. θ Savgol smoothing across time, per-joint, per-axis.
        thetas_smooth = savgol_axis(
            thetas, window=savgol_window, order=savgol_order, axis=0
        ).astype(np.float32)

        # 5. Foot-anchored translation per-frame.
        pose_path = pose_dir / f"{player_id}_pose.json"
        with pose_path.open() as fh:
            pose_data = json.load(fh)
        pose_by_frame = {int(p["frame"]): p for p in pose_data["frames"]}

        root_t = np.zeros((len(frame_indices), 3), dtype=float)
        confidence = np.zeros(len(frame_indices), dtype=float)
        last_anchored: np.ndarray | None = None
        for i, fi in enumerate(frame_indices):
            fi_int = int(fi)
            if not camera_present[i] or fi_int not in pose_by_frame:
                # No camera or no 2D pose — leave translation zero, confidence zero.
                continue
            K = per_frame_K[fi_int]
            R = per_frame_R[fi_int]
            kp = np.asarray(pose_by_frame[fi_int]["keypoints"], dtype=float)
            left = kp[_COCO_LEFT_ANKLE]
            right = kp[_COCO_RIGHT_ANKLE]
            ankle_conf = float(min(left[2], right[2]))
            if ankle_conf < _ANKLE_CONF_MIN:
                # Low-confidence keypoints: hold last anchor (avoids teleport)
                # and flag the frame with attenuated confidence per the spec
                # error-handling philosophy ("flag, don't substitute").
                if last_anchored is not None:
                    root_t[i] = last_anchored
                confidence[i] = ankle_conf  # propagate the low score
                continue
            ankle_uv = (
                (left[0] + right[0]) / 2.0,
                (left[1] + right[1]) / 2.0,
            )
            try:
                foot_world = ankle_ray_to_pitch(
                    ankle_uv, K=K, R=R, t=t_world, plane_z=_FOOT_PLANE_Z
                )
            except ValueError:
                # Ray parallel to ground — skip this frame.
                if last_anchored is not None:
                    root_t[i] = last_anchored
                confidence[i] = 0.0
                continue
            root_t[i] = anchor_translation(
                foot_world, _FOOT_IN_ROOT, root_R_pitch[i]
            )
            last_anchored = root_t[i]
            joint_conf_min = float(joint_conf[i].min()) if joint_conf.size else 0.0
            confidence[i] = float(min(ankle_conf, joint_conf_min))

        # 6. Ground-snap z when the avatar is approximately stationary.
        root_t[:, 2] = ground_snap_z(
            root_t[:, 2], velocity_threshold=ground_snap_velocity
        )

        track = SmplWorldTrack(
            player_id=str(player_id),
            frames=frame_indices.astype(np.int64),
            betas=betas.astype(np.float32),
            thetas=thetas_smooth.astype(np.float32),
            root_R=root_R_pitch.astype(np.float32),
            root_t=root_t.astype(np.float32),
            confidence=confidence.astype(np.float32),
        )
        track.save(out_dir / f"{player_id}_smpl_world.npz")

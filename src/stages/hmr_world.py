"""HMR-in-pitch-frame stage.

Per-player monocular SMPL reconstruction expressed in pitch-world coords.

For each player track in ``output/tracks/{shot_id}_tracks.json``:
1. Run GVHMR over the track to obtain per-frame SMPL params in the camera
   frame (root rotation, pose, shape) plus COCO-17 2D keypoints from
   GVHMR's internal ViTPose-Huge.
2. Median-aggregate the (per-frame-noisy) shape parameters.
3. Convert root rotation from camera frame to pitch frame via the calibrated
   camera extrinsic, then SLERP-smooth.
4. Savgol-smooth the per-joint axis-angle pose.
5. Compute per-frame translation by ankle-anchoring: project the 2D ankle
   midpoint (from GVHMR's kp2d) to the pitch ground plane (z = 0.05 m) and
   back-solve the root translation that places the foot exactly there.
6. Ground-snap z when the avatar is roughly stationary.

Outputs per player:
- ``output/hmr_world/{player_id}_smpl_world.npz`` — SmplWorldTrack
- ``output/hmr_world/{player_id}_kp2d.json``      — COCO-17 keypoints
  (consumed by the dashboard 2D-overlay panel; same schema the legacy
  pose_2d stage used to emit).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from src.pipeline.base import BaseStage

logger = logging.getLogger(__name__)
from src.schemas.camera_track import CameraTrack
from src.schemas.smpl_world import SmplWorldTrack
from src.schemas.tracks import TracksResult
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

# Foot offset relative to the SMPL root, in the body's local (SMPL
# canonical, y-up) frame. ``root_R_pitch`` rotates the body from this
# y-up local frame straight into pitch z-up world (see the docstring
# in ``smpl_pitch_transform``), so foot-below-root is along the body's
# local -y, not pitch -z.
#
# (Decision D9 in the implementation log called this offset
# ``(0, 0, -0.95)`` because the previous transform had a misnamed bridge
# matrix that left the body-local frame coincidentally z-up. With the
# bridge fixed, the offset is now in correct SMPL canonical convention.)
_FOOT_IN_ROOT = np.array([0.0, -0.95, 0.0], dtype=float)

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
        out_dir = self.output_dir / "hmr_world"
        out_dir.mkdir(parents=True, exist_ok=True)

        camera_track = CameraTrack.load(camera_path)
        per_frame_K = {f.frame: np.array(f.K, dtype=float) for f in camera_track.frames}
        per_frame_R = {f.frame: np.array(f.R, dtype=float) for f in camera_track.frames}
        # Per-frame t when available (current camera stage always writes
        # it). Fall back to clip-shared t_world for legacy tracks. Per-
        # frame t is essential for static-camera clips where t varies
        # with R while the body centre stays put.
        t_world_fallback = np.array(camera_track.t_world, dtype=float)
        per_frame_t: dict[int, np.ndarray] = {}
        for f in camera_track.frames:
            if f.t is not None:
                per_frame_t[f.frame] = np.array(f.t, dtype=float)
            else:
                per_frame_t[f.frame] = t_world_fallback
        distortion = camera_track.distortion

        min_track_frames = int(cfg.get("min_track_frames", 10))
        savgol_window = int(cfg.get("theta_savgol_window", 11))
        savgol_order = int(cfg.get("theta_savgol_order", 2))
        slerp_w = int(cfg.get("root_slerp_window", 5))
        ground_snap_velocity = float(cfg.get("ground_snap_velocity", 0.1))
        # Translation jitter dampener — Savgol-smoothed across time. Same
        # signal source (per-frame foot-anchor ray-cast) carries any
        # camera-tracking jitter through to root_t; a 5-frame window
        # cancels it without flattening sprint accelerations.
        root_t_savgol_window = int(cfg.get("root_t_savgol_window", 5))
        root_t_savgol_order = int(cfg.get("root_t_savgol_order", 2))
        # Constant tilt-correction (degrees) for the monocular HMR's
        # lean-away-from-camera bias. Rotates root_R_pitch around the
        # horizontal axis perpendicular to camera-to-player so the body
        # stands "lean_correction_deg" further toward the camera.
        # 0 disables. Sign convention: positive = tilt body toward camera.
        lean_correction_deg = float(cfg.get("lean_correction_deg", 0.0))

        # Build per-player frame lists by walking each shot's TracksResult.
        # Tracks share a player_id (assigned via the dashboard's annotation
        # UI: name/split/merge); fallback to track_id when the operator
        # hasn't annotated yet so the stage still runs against raw tracking
        # output. Tracks named "ignore" are dropped entirely.
        groups: dict[str, list[tuple[int, tuple[int, int, int, int]]]] = {}
        group_shot: dict[str, str] = {}
        if not track_dir.exists():
            return
        for tracks_path in sorted(track_dir.glob("*_tracks.json")):
            try:
                tr = TracksResult.load(tracks_path)
            except Exception:
                continue
            for track in tr.tracks:
                if track.class_name not in ("player", "goalkeeper"):
                    continue
                if track.player_name == "ignore":
                    continue
                pid = track.player_id or track.track_id
                if pid not in groups:
                    groups[pid] = []
                    group_shot[pid] = tr.shot_id
                groups[pid].extend(
                    (int(f.frame), tuple(int(x) for x in f.bbox))
                    for f in track.frames
                )

        # Sort players for stable ordering across runs (deterministic
        # resume: the same partial order each time means the operator
        # can predict which players come next).
        ordered = sorted(groups.items(), key=lambda kv: kv[0])
        total = len(ordered)
        cached = sum(
            1 for pid, _ in ordered
            if (out_dir / f"{pid}_smpl_world.npz").exists()
        )
        to_process = total - cached
        if total == 0:
            print("[hmr_world] no tracks to process")
            return
        print(
            f"[hmr_world] {total} player(s): "
            f"{cached} cached on disk, {to_process} to process"
        )

        # Build one estimator for the whole stage. GVHMR + ViTPose-Huge +
        # HMR2.0 ViT-Huge + SMPLX load is 30-60s; without this, every
        # player paid that cost (the previous run_on_track constructed a
        # fresh estimator per call). Lazy: only build when there's
        # something to process — an all-cached run is still torch-free.
        estimator = None
        if to_process > 0:
            from src.utils.gvhmr_estimator import GVHMREstimator

            estimator = GVHMREstimator(
                checkpoint=str(cfg.get("checkpoint", "")),
                device=str(cfg.get("device", "auto")),
            )

        run_start = time.time()
        elapsed_per_player: list[float] = []
        for i, (player_id, frames) in enumerate(ordered, start=1):
            frames = sorted(set(frames), key=lambda x: x[0])
            t0 = time.time()
            status = self._process_player(
                player_id=player_id,
                shot_id=group_shot[player_id],
                track_frames=frames,
                out_dir=out_dir,
                cfg=cfg,
                per_frame_K=per_frame_K,
                per_frame_R=per_frame_R,
                per_frame_t=per_frame_t,
                distortion=distortion,
                min_track_frames=min_track_frames,
                savgol_window=savgol_window,
                savgol_order=savgol_order,
                slerp_w=slerp_w,
                ground_snap_velocity=ground_snap_velocity,
                root_t_savgol_window=root_t_savgol_window,
                root_t_savgol_order=root_t_savgol_order,
                lean_correction_deg=lean_correction_deg,
                estimator=estimator,
            )
            dt = time.time() - t0
            if status == "ran":
                elapsed_per_player.append(dt)
                avg = sum(elapsed_per_player) / len(elapsed_per_player)
                remaining = sum(
                    1 for pid, _ in ordered[i:]
                    if not (out_dir / f"{pid}_smpl_world.npz").exists()
                )
                eta = avg * remaining
                print(
                    f"[hmr_world] ({i}/{total}) {player_id} done in "
                    f"{_fmt_duration(dt)}  "
                    f"(avg {_fmt_duration(avg)}/player, ~{_fmt_duration(eta)} remaining)"
                )
            elif status == "cached":
                print(f"[hmr_world] ({i}/{total}) {player_id} cached, skipping")
            elif status == "too_short":
                print(
                    f"[hmr_world] ({i}/{total}) {player_id} skipped "
                    f"({len(frames)} < min_track_frames={min_track_frames})"
                )

        print(
            f"[hmr_world] done — total wall {_fmt_duration(time.time() - run_start)}"
        )

    def _process_player(
        self,
        *,
        player_id: str,
        shot_id: str,
        track_frames: list[tuple[int, tuple[int, int, int, int]]],
        out_dir: Path,
        cfg: dict,
        per_frame_K: dict[int, np.ndarray],
        per_frame_R: dict[int, np.ndarray],
        per_frame_t: dict[int, np.ndarray],
        distortion: tuple[float, float],
        min_track_frames: int,
        savgol_window: int,
        savgol_order: int,
        slerp_w: int,
        ground_snap_velocity: float,
        root_t_savgol_window: int,
        root_t_savgol_order: int,
        lean_correction_deg: float,
        estimator: object | None = None,
    ) -> str:
        """Process one player. Returns one of:
        - ``"too_short"`` — track had fewer than ``min_track_frames`` frames
        - ``"cached"`` — output already on disk, skipped GVHMR
        - ``"ran"`` — GVHMR ran and a fresh SmplWorldTrack was written
        """
        if len(track_frames) < min_track_frames:
            return "too_short"

        # Per-player resume: if the .npz is already on disk, skip GVHMR
        # for this player. CPU GVHMR is ~5 min/player, so a kill-and-
        # resume across 20+ players otherwise repeats hours of work. The
        # dashboard's Re-run Stage button still wipes the directory before
        # invoking, so an explicit re-run from the UI is unaffected.
        out_path = out_dir / f"{player_id}_smpl_world.npz"
        if out_path.exists():
            return "cached"

        # Announce up front — this player is going to take minutes on CPU.
        print(
            f"[hmr_world] {player_id} ({shot_id}) running — {len(track_frames)} frames…",
            flush=True,
        )

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
            estimator=estimator,
        )
        thetas = np.asarray(hmr_out["thetas"])             # (N, 24, 3)
        betas_all = np.asarray(hmr_out["betas"])           # (N, 10)
        root_R_cam = np.asarray(hmr_out["root_R_cam"])     # (N, 3, 3)
        joint_conf = np.asarray(hmr_out["joint_confidence"])  # (N, 24)
        kp2d = np.asarray(hmr_out["kp2d"])                 # (N, 17, 3)

        # GVHMR's body_pose axis-angles, when fed through our viewer's
        # standard right-multiply FK chain (rot[i] = rot[par] @ Rl[i]),
        # render every joint with REVERSED rotation: knees hyperextend,
        # spine arches backward, arms swing up and behind. Inverting
        # each axis-angle vector (negating all three components, which
        # equivalently transposes the corresponding rotation matrix)
        # produces correct anatomical motion. Confirmed empirically via
        # a per-joint pose-convention selector in the viewer.
        #
        # GVHMR's own SMPL FK in third_party/gvhmr/.../smplx_lite.py:267
        # is mathematically the same chain as ours, so the underlying
        # cause is some implicit convention we haven't fully isolated
        # (handed differently between SMPL releases or between PyTorch3D
        # axis-angle and our JS Rodrigues). The fix is small and
        # local; investigating the upstream root cause is an open task.
        #
        # The historical "thetas[:, 1:22, 1:3] *= -1" (180°-around-X
        # conjugation) was a partial fix that handled some axes but
        # left yaw/roll reversed; the full-vector negation here covers
        # all three axes uniformly.
        thetas[:, 1:22, :] *= -1.0

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

        # 5. Foot-anchored translation per-frame using GVHMR's internal
        # ViTPose kp2d (one entry per track frame, aligned with frame_indices).
        root_t = np.zeros((len(frame_indices), 3), dtype=float)
        confidence = np.zeros(len(frame_indices), dtype=float)
        last_anchored: np.ndarray | None = None
        for i, fi in enumerate(frame_indices):
            fi_int = int(fi)
            if not camera_present[i]:
                # No camera — leave translation zero, confidence zero.
                continue
            K = per_frame_K[fi_int]
            R = per_frame_R[fi_int]
            t = per_frame_t[fi_int]
            kp = kp2d[i]
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
                    ankle_uv, K=K, R=R, t=t,
                    plane_z=_FOOT_PLANE_Z, distortion=distortion,
                )
            except ValueError:
                # Ray parallel to ground — skip this frame.
                if last_anchored is not None:
                    root_t[i] = last_anchored
                confidence[i] = 0.0
                continue
            # Lean-correction: rotate root_R_pitch[i] around the horizontal
            # axis perpendicular to (camera → player) by a fixed angle to
            # counter monocular HMR's away-from-camera bias. Applied
            # before the foot-anchor translation so the foot stays glued
            # to its detected pitch position under the corrected R.
            if lean_correction_deg != 0.0:
                cam_centre = -R.T @ t
                v_horiz = np.array(
                    [foot_world[0] - cam_centre[0],
                     foot_world[1] - cam_centre[1],
                     0.0],
                    dtype=float,
                )
                v_norm = float(np.linalg.norm(v_horiz))
                if v_norm > 1e-6:
                    v_horiz /= v_norm
                    z_up = np.array([0.0, 0.0, 1.0])
                    lean_axis = np.cross(v_horiz, z_up)
                    lean_axis_norm = float(np.linalg.norm(lean_axis))
                    if lean_axis_norm > 1e-6:
                        lean_axis /= lean_axis_norm
                        ang = np.deg2rad(lean_correction_deg)
                        K_x = np.array([
                            [0.0, -lean_axis[2], lean_axis[1]],
                            [lean_axis[2], 0.0, -lean_axis[0]],
                            [-lean_axis[1], lean_axis[0], 0.0],
                        ])
                        # Rodrigues' rotation matrix.
                        correction_R = (
                            np.eye(3)
                            + np.sin(ang) * K_x
                            + (1 - np.cos(ang)) * K_x @ K_x
                        )
                        root_R_pitch[i] = correction_R @ root_R_pitch[i]
            root_t[i] = anchor_translation(
                foot_world, _FOOT_IN_ROOT, root_R_pitch[i]
            )
            last_anchored = root_t[i]
            joint_conf_min = float(joint_conf[i].min()) if joint_conf.size else 0.0
            confidence[i] = float(min(ankle_conf, joint_conf_min))

        # 6. (No ground snap.) The previous ``ground_snap_z`` post-process
        # halved root_t.z every frame whose per-frame velocity was below
        # threshold — which is every frame for a stationary or slowly-
        # moving player. That collapsed the pelvis toward z=0 (so the
        # avatar's feet ended up below the pitch). The foot-anchor
        # ray-cast above already constrains the pelvis position
        # consistently with the ankle keypoint, so no extra snap is
        # needed. ``ground_snap_velocity`` is kept in the signature for
        # backwards-compat but is now ignored.
        _ = ground_snap_velocity

        # 7. Translation jitter dampener. Camera-tracking jitter feeds
        # directly into root_t via the per-frame foot-anchor ray-cast;
        # a Savgol smoother across time absorbs sub-frame noise without
        # flattening the player's actual motion. Skip when the track is
        # too short to apply the window or the smoother is disabled.
        if (
            root_t_savgol_window > 1
            and root_t.shape[0] >= root_t_savgol_window
        ):
            root_t = savgol_axis(
                root_t,
                window=root_t_savgol_window,
                order=root_t_savgol_order,
                axis=0,
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

        # Side-output: COCO-17 keypoints for the dashboard 2D-overlay panel.
        # Same JSON schema the legacy pose_2d stage emitted, so the existing
        # renderer in src/web/static/index.html can consume it unchanged.
        kp2d_payload = {
            "player_id": str(player_id),
            "shot_id": shot_id,
            "frames": [
                {"frame": int(fi), "keypoints": kp2d[i].tolist()}
                for i, fi in enumerate(frame_indices)
            ],
        }
        (out_dir / f"{player_id}_kp2d.json").write_text(json.dumps(kp2d_payload))
        return "ran"


def _fmt_duration(seconds: float) -> str:
    """Compact m:ss / h:mm:ss for the progress lines."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m = int(seconds // 60); s = int(seconds - m * 60)
        return f"{m}m{s:02d}s"
    h = int(seconds // 3600); m = int((seconds - h * 3600) // 60)
    return f"{h}h{m:02d}m"

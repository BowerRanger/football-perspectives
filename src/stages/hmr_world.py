"""HMR-in-pitch-frame stage.

Per-player, **per-shot** monocular SMPL reconstruction expressed in
pitch-world coords. Each shot is solved independently — when the same
``player_id`` appears in multiple shots (e.g. after Merge by Name), one
output file is written per shot. Cross-shot data is *not* combined; a
later convergence stage is responsible for that if/when needed.

For each (shot_id, player_id) pair in ``output/tracks/{shot_id}_tracks.json``:
1. Run GVHMR over the track to obtain per-frame SMPL params in the camera
   frame (root rotation, pose, shape) plus COCO-17 2D keypoints from
   GVHMR's internal ViTPose-Huge.
2. Median-aggregate the (per-frame-noisy) shape parameters.
3. Convert root rotation from camera frame to pitch frame via the
   calibrated camera extrinsic for *this shot*, then SLERP-smooth.
4. Savgol-smooth the per-joint axis-angle pose.
5. Compute per-frame translation by ankle-anchoring: project the 2D ankle
   midpoint (from GVHMR's kp2d) to the pitch ground plane (z = 0.05 m) and
   back-solve the root translation that places the foot exactly there.

Outputs per (shot, player) pair:
- ``output/hmr_world/{shot_id}__{player_id}_smpl_world.npz`` — SmplWorldTrack
- ``output/hmr_world/{shot_id}__{player_id}_kp2d.json``      — COCO-17 keypoints
  (consumed by the dashboard 2D-overlay panel; same schema the legacy
  pose_2d stage used to emit).

The ``__`` separator delimits shot_id from player_id at the filename
level. Both substrings are constrained to ``[A-Za-z0-9_-]`` upstream;
parsers should ``rsplit("__", 1)`` to recover the player_id rather than
splitting on ``_``.
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
from src.utils.smpl_skeleton import SMPL_REST_JOINTS_YUP
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

# Ankle-joint offset relative to the SMPL root (pelvis), in the body's
# local (SMPL canonical, y-up) frame. ``root_R_pitch`` rotates the body
# from this y-up local frame straight into pitch z-up world (see the
# docstring in ``smpl_pitch_transform``), so ankle-below-root is along
# the body's local -y, not pitch -z.
#
# Derived from the SMPL canonical rest-pose joint table (mean betas):
# left/right ankle indices 7/8 are at y=-0.882 m, with a small forward
# z offset and ±x lateral offset. Averaging the two and zeroing the
# lateral component anchors the root over the foot-midpoint pixel that
# we ray-cast (see ``_FOOT_PLANE_Z`` below).
#
# (Decision D9 in the implementation log called this offset
# ``(0, 0, -0.95)`` and a later revision used ``(0, -0.95, 0)``. Both
# treated the constant as the root-to-*sole* distance, which left
# players floating ~7 cm above the pitch because the anchored 2D
# keypoint and the ray-cast plane both refer to the *ankle*, not the
# sole. Using the SMPL canonical ankle position resolves the mismatch
# without per-shape forward-kinematics, accurate to ~1-2 cm across
# typical adult beta variation.)
_ANKLE_IN_ROOT = 0.5 * (
    SMPL_REST_JOINTS_YUP[7] + SMPL_REST_JOINTS_YUP[8]
).astype(float)
_ANKLE_IN_ROOT[0] = 0.0  # zero lateral: root sits over the foot midpoint

# Pitch-frame z of the ankle keypoint when standing on the turf. ViTPose
# annotates the lateral malleolus (the bony ankle bump), which sits a
# few centimetres above the boot sole — the small positive offset both
# matches that anatomy and keeps a near-vertical ray from grazing-
# intersecting the ground plane.
_FOOT_PLANE_Z = 0.05


_OUTPUT_SEPARATOR = "__"


def _output_key(shot_id: str, player_id: str) -> str:
    """Filename-safe key joining ``shot_id`` and ``player_id``.

    The pipeline guarantees both substrings only contain ``[A-Za-z0-9_-]``
    (see ``_sanitise_shot_id`` and the server-side player_id validator),
    so a literal ``__`` separator unambiguously splits the pair on
    ``rsplit("__", 1)``.
    """
    return f"{shot_id}{_OUTPUT_SEPARATOR}{player_id}"


def _wipe_legacy_outputs(out_dir: Path) -> int:
    """Delete pre-multi-shot ``hmr_world`` artefacts.

    Files written before the per-shot rename use ``{player_id}_smpl_world.npz``
    (no ``__`` separator). They were solved against whichever camera the
    stage saw first for that player_id, which is wrong in multi-shot
    mode. We delete them on the first new-scheme run so the user can
    cleanly rebuild without stale combined animations leaking into the
    viewer.
    """
    if not out_dir.exists():
        return 0
    removed = 0
    for path in list(out_dir.glob("*_smpl_world.npz")):
        if _OUTPUT_SEPARATOR not in path.stem:
            path.unlink()
            removed += 1
    for path in list(out_dir.glob("*_kp2d.json")):
        if _OUTPUT_SEPARATOR not in path.stem:
            path.unlink()
            removed += 1
    if removed:
        logger.info(
            "[hmr_world] wiped %d legacy single-shot artefact(s) — they"
            " predate the per-shot output rename",
            removed,
        )
    return removed


class HmrWorldStage(BaseStage):
    name = "hmr_world"

    def is_complete(self) -> bool:
        out = self.output_dir / "hmr_world"
        if not out.exists():
            return False
        # Only count new-scheme files. A directory full of legacy combined
        # files shouldn't flip the stage green when the new code would
        # rebuild them all.
        return any(
            _OUTPUT_SEPARATOR in p.stem for p in out.glob("*_smpl_world.npz")
        )

    def run(self) -> None:
        from src.schemas.shots import ShotsManifest

        cfg = self.config.get("hmr_world", {})
        track_dir = self.output_dir / "tracks"
        out_dir = self.output_dir / "hmr_world"
        out_dir.mkdir(parents=True, exist_ok=True)
        _wipe_legacy_outputs(out_dir)

        # Load every shot's camera_track separately. Each player's
        # animation is solved against the camera track of the shot they
        # were detected in (group_shot[pid] → shot_id → camera).
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        camera_tracks_by_shot: dict[str, CameraTrack] = {}
        per_frame_K_by_shot: dict[str, dict[int, np.ndarray]] = {}
        per_frame_R_by_shot: dict[str, dict[int, np.ndarray]] = {}
        per_frame_t_by_shot: dict[str, dict[int, np.ndarray]] = {}
        distortion_by_shot: dict[str, tuple[float, float]] = {}

        def _build_per_frame(shot_key: str, cam: CameraTrack) -> None:
            camera_tracks_by_shot[shot_key] = cam
            per_frame_K_by_shot[shot_key] = {
                f.frame: np.array(f.K, dtype=float) for f in cam.frames
            }
            per_frame_R_by_shot[shot_key] = {
                f.frame: np.array(f.R, dtype=float) for f in cam.frames
            }
            t_fb = np.array(cam.t_world, dtype=float)
            per_frame_t_by_shot[shot_key] = {
                f.frame: (np.array(f.t, dtype=float) if f.t is not None else t_fb)
                for f in cam.frames
            }
            distortion_by_shot[shot_key] = cam.distortion

        if manifest_path.exists():
            manifest = ShotsManifest.load(manifest_path)
            for shot in manifest.shots:
                p = self.output_dir / "camera" / f"{shot.id}_camera_track.json"
                if not p.exists():
                    logger.warning(
                        "hmr_world skipping shot %s — no camera track at %s",
                        shot.id, p,
                    )
                    continue
                _build_per_frame(shot.id, CameraTrack.load(p))

        # Legacy single-shot fallback: if no per-shot files but a singular
        # one exists, register it under shot_id="" so downstream lookup
        # still works (with the same key when group_shot lookup misses).
        if not camera_tracks_by_shot:
            legacy = self.output_dir / "camera" / "camera_track.json"
            if legacy.exists():
                _build_per_frame("", CameraTrack.load(legacy))
        logger.info(
            "[hmr_world] camera tracks loaded for %d shot(s): %s",
            len(camera_tracks_by_shot), list(camera_tracks_by_shot.keys()),
        )

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

        if not track_dir.exists():
            return
        groups = self._build_player_groups()

        shot_filter = getattr(self, "shot_filter", None)
        player_filter = getattr(self, "player_filter", None)
        if shot_filter is not None:
            groups = {
                k: v for k, v in groups.items() if k[0] == shot_filter
            }
        if player_filter is not None:
            groups = {
                k: v for k, v in groups.items() if k[1] == player_filter
            }

        # Sort by (shot_id, player_id) for stable ordering across runs
        # (deterministic resume: the same partial order each time means
        # the operator can predict which players come next).
        ordered = sorted(groups.items(), key=lambda kv: kv[0])
        total = len(ordered)
        cached = sum(
            1 for (sid, pid), _ in ordered
            if (out_dir / f"{_output_key(sid, pid)}_smpl_world.npz").exists()
        )
        to_process = total - cached
        if total == 0:
            filter_note = ""
            if shot_filter or player_filter:
                filter_note = (
                    f" matching shot={shot_filter!r} player={player_filter!r}"
                )
            print(f"[hmr_world] no tracks to process{filter_note}")
            return
        print(
            f"[hmr_world] {total} (shot, player) group(s): "
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
        for i, ((shot_id_for_pid, player_id), frames) in enumerate(ordered, start=1):
            frames = sorted(set(frames), key=lambda x: x[0])
            t0 = time.time()
            shot_key = (
                shot_id_for_pid if shot_id_for_pid in camera_tracks_by_shot else ""
            )
            status = self._process_player(
                player_id=player_id,
                shot_id=shot_id_for_pid,
                track_frames=frames,
                out_dir=out_dir,
                cfg=cfg,
                per_frame_K=per_frame_K_by_shot.get(shot_key, {}),
                per_frame_R=per_frame_R_by_shot.get(shot_key, {}),
                per_frame_t=per_frame_t_by_shot.get(shot_key, {}),
                distortion=distortion_by_shot.get(shot_key, (0.0, 0.0)),
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
            label = _output_key(shot_id_for_pid, player_id)
            dt = time.time() - t0
            if status == "ran":
                elapsed_per_player.append(dt)
                avg = sum(elapsed_per_player) / len(elapsed_per_player)
                remaining = sum(
                    1 for (sid2, pid2), _ in ordered[i:]
                    if not (out_dir / f"{_output_key(sid2, pid2)}_smpl_world.npz").exists()
                )
                eta = avg * remaining
                print(
                    f"[hmr_world] ({i}/{total}) {label} done in "
                    f"{_fmt_duration(dt)}  "
                    f"(avg {_fmt_duration(avg)}/player, ~{_fmt_duration(eta)} remaining)"
                )
            elif status == "cached":
                print(f"[hmr_world] ({i}/{total}) {label} cached, skipping")
            elif status == "too_short":
                print(
                    f"[hmr_world] ({i}/{total}) {label} skipped "
                    f"({len(frames)} < min_track_frames={min_track_frames})"
                )

        print(
            f"[hmr_world] done — total wall {_fmt_duration(time.time() - run_start)}"
        )

    def _build_player_groups(
        self,
    ) -> dict[tuple[str, str], list[tuple[int, tuple[int, int, int, int]]]]:
        """Walk every {shot_id}_tracks.json and group track frames by
        ``(shot_id, player_id)`` — never combining across shots. The
        same ``player_id`` appearing in two shots produces two separate
        groups (and therefore two separate output files); a later
        convergence stage can choose how to fuse them if needed.

        Unannotated tracks (no player_id, no player_name) get a shot-
        prefixed pid (``"{shot_id}_T{track_id}"``) so different physical
        players never collapse into one group when the operator hasn't
        named them yet.
        """
        groups: dict[tuple[str, str], list[tuple[int, tuple[int, int, int, int]]]] = {}
        track_dir = self.output_dir / "tracks"
        if not track_dir.exists():
            return groups
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
                pid = (
                    track.player_id
                    or (
                        f"{tr.shot_id}_T{track.track_id}"
                        if track.track_id else None
                    )
                )
                if pid is None:
                    continue
                key = (tr.shot_id, pid)
                if key not in groups:
                    groups[key] = []
                groups[key].extend(
                    (int(f.frame), tuple(int(x) for x in f.bbox))
                    for f in track.frames
                )
        return groups

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
        video_path = self.output_dir / "shots" / f"{shot_id}.mp4"
        return process_player(
            player_id=player_id,
            shot_id=shot_id,
            track_frames=track_frames,
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
            video_path=video_path,
        )


def process_player(
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
    video_path: Path,
    estimator: object | None = None,
) -> str:
    """Process one player. Returns one of:
    - ``"too_short"`` — track had fewer than ``min_track_frames`` frames
    - ``"cached"`` — output already on disk, skipped GVHMR
    - ``"ran"`` — GVHMR ran and a fresh SmplWorldTrack was written

    ``video_path`` is the absolute path to the shot's MP4 clip on local
    disk — ``output/shots/{shot_id}.mp4``.
    """
    if len(track_frames) < min_track_frames:
        return "too_short"

    out_key = _output_key(shot_id, player_id)
    # Per-(shot, player) resume: if the .npz is already on disk, skip
    # GVHMR. CPU GVHMR is ~5 min/player, so a kill-and-resume across
    # 20+ players otherwise repeats hours of work. The dashboard's
    # Re-run Stage button still wipes the directory before invoking,
    # so an explicit re-run from the UI is unaffected.
    out_path = out_dir / f"{out_key}_smpl_world.npz"
    if out_path.exists():
        return "cached"

    # Announce up front — this player is going to take minutes on CPU.
    print(
        f"[hmr_world] {out_key} running — {len(track_frames)} frames…",
        flush=True,
    )

    # 1. GVHMR per track (lazy import — heavy dependency).
    from src.utils.gvhmr_estimator import run_on_track

    if not video_path.exists():
        raise RuntimeError(
            f"hmr_world: shot clip not found at {video_path} — run "
            "prepare_shots for this shot first"
        )

    # Build per-track-frame K array for GVHMR. The default
    # ``estimate_K(w, h)`` inside GVHMR assumes ~60° FOV which under-
    # estimates focal length for broadcast telephoto and biases the
    # predicted body to lean away from the camera. Passing the
    # calibrated per-frame K (from camera_track) eliminates that
    # mismatch. Frames missing from ``per_frame_K`` fall back to the
    # shot's median K so GVHMR receives a dense array.
    gvhmr_K: np.ndarray | None = None
    if per_frame_K:
        K_values = np.stack(list(per_frame_K.values()))
        K_median = np.median(K_values, axis=0)
        gvhmr_K = np.stack(
            [per_frame_K.get(int(fi), K_median) for fi, _ in track_frames]
        ).astype(np.float32)

    hmr_out = run_on_track(
        track_frames=track_frames,
        video_path=video_path,
        checkpoint=Path(cfg.get("checkpoint", "")),
        device=str(cfg.get("device", "auto")),
        batch_size=int(cfg.get("batch_size", 16)),
        max_sequence_length=int(cfg.get("max_sequence_length", 120)),
        estimator=estimator,
        per_frame_K=gvhmr_K,
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
    # Frames where a fresh ankle ray-cast succeeded carry their solved
    # ``root_t``; low-confidence frames hold the last good anchor;
    # camera-missing frames stay at zero (and zero confidence). The
    # downstream ``refined_poses`` stage trims the leading/trailing
    # un-anchored span using ``confidence`` as the anchored signal.
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
            foot_world, _ANKLE_IN_ROOT, root_R_pitch[i]
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
        shot_id=shot_id,
    )
    track.save(out_dir / f"{out_key}_smpl_world.npz")

    # Side-output: COCO-17 keypoints for the dashboard 2D-overlay panel.
    # Same JSON schema the legacy pose_2d stage emitted; the renderer
    # in src/web/static/index.html consumes it via the per-shot
    # /hmr_world/kp2d_* endpoints.
    kp2d_payload = {
        "player_id": str(player_id),
        "shot_id": shot_id,
        "frames": [
            {"frame": int(fi), "keypoints": kp2d[i].tolist()}
            for i, fi in enumerate(frame_indices)
        ],
    }
    (out_dir / f"{out_key}_kp2d.json").write_text(json.dumps(kp2d_payload))
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

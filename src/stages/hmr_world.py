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
5. Apply the monocular lean-correction to root_R (see
   ``_apply_lean_correction``), then anchor the per-frame root
   translation (see ``anchor_root_translation``, shared with
   ``scripts/reanchor_hmr_world.py``): ``hmr_world.anchor_mode``
   ``"contact"`` (default) ray-casts each ankle to the pitch ground plane
   to detect per-foot stance spans and pins the root so the planted foot
   stops sliding; ``"ankle_mid"`` reproduces the legacy single-offset
   ankle-midpoint ray-cast anchor.

Outputs per (shot, player) pair:
- ``output/hmr_world/{shot_id}__{player_id}_smpl_world.npz`` — SmplWorldTrack
- ``output/hmr_world/{shot_id}__{player_id}_kp2d.json``      — COCO-17 keypoints
  (consumed by the dashboard 2D-overlay panel; same schema the legacy
  pose_2d stage used to emit).
- ``output/hmr_world/{shot_id}__{player_id}_foot_contacts.json`` — per-foot
  stance spans (``anchor_mode: contact`` only; see
  ``src/schemas/foot_contacts.py``).

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
from src.schemas.foot_contacts import save_foot_contacts
from src.schemas.smpl_world import SmplWorldTrack
from src.schemas.tracks import TracksResult
from src.utils.foot_anchor import ankle_ray_to_pitch, anchor_translation
from src.utils.foot_contact import FootContacts, detect_contacts
from src.utils.foot_lock import solve_root_with_pins
from src.utils.smpl_pitch_transform import smpl_root_in_pitch_frame
from src.utils.smpl_skeleton import (
    SMPL_REST_JOINTS_YUP,
    beta_adjusted_rest_joints,
    compute_canonical_joints_batch,
    load_smpl_neutral_model,
)
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
            for shot in manifest.active_shots():
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
                extractor_device=str(cfg.get("extractor_device", "cpu")),
            )

        run_start = time.time()
        elapsed_per_player: list[float] = []
        for i, ((shot_id_for_pid, player_id), frames) in enumerate(ordered, start=1):
            frames = sorted(set(frames), key=lambda x: x[0])
            t0 = time.time()
            shot_key = (
                shot_id_for_pid if shot_id_for_pid in camera_tracks_by_shot else ""
            )
            shot_fps = (
                camera_tracks_by_shot[shot_key].fps
                if shot_key in camera_tracks_by_shot else 25.0
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
                fps=float(shot_fps),
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
        fps: float = 25.0,
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
            fps=fps,
            estimator=estimator,
            video_path=video_path,
        )


def build_track_camera_R(
    track_frames: list[tuple[int, tuple[int, int, int, int]]],
    per_frame_R: dict[int, np.ndarray],
) -> np.ndarray | None:
    """Build a dense ``(N, 3, 3)`` world-to-camera rotation array aligned
    to ``track_frames`` order, for GVHMR's calibrated-camera-R path
    (``run_on_track``'s ``per_frame_R`` / ``estimate_sequence``'s
    ``R_w2c_per_frame``).

    ``per_frame_R`` (the camera_track's per-frame R, same dict already
    used a few lines below to convert root rotation into pitch frame)
    rarely covers every track frame exactly — camera anchors and
    propagation can leave small per-shot gaps. Missing frames are filled
    from the NEAREST available frame's R: the previous frame when one has
    already been seen, otherwise the next available frame (for a leading
    gap before any camera data). Rotations are never averaged — blending
    two rotation matrices isn't a rotation.

    Returns ``None`` when ``per_frame_R`` is empty or none of its keys
    match any frame in ``track_frames`` (the caller then falls back to
    GVHMR's own SimpleVO estimate).

    Factored out as a module-level helper (rather than inlined in
    ``process_player``) so ``scripts/bench_gvhmr_inference.py`` can build
    the identical dense-R array outside the stage.
    """
    if not per_frame_R:
        return None
    frame_indices = [int(fi) for fi, _ in track_frames]
    n = len(frame_indices)
    resolved: list[np.ndarray | None] = [None] * n
    for i, fi in enumerate(frame_indices):
        R = per_frame_R.get(fi)
        if R is not None:
            resolved[i] = np.asarray(R, dtype=np.float32)
    if all(r is None for r in resolved):
        return None

    # Forward-fill: propagate each frame's R to subsequent missing frames
    # (nearest previous available frame).
    last: np.ndarray | None = None
    for i in range(n):
        if resolved[i] is not None:
            last = resolved[i]
        elif last is not None:
            resolved[i] = last

    # Backward-fill any remaining leading gap (no previous frame existed
    # yet) from the next available frame.
    nxt: np.ndarray | None = None
    for i in range(n - 1, -1, -1):
        if resolved[i] is not None:
            nxt = resolved[i]
        elif nxt is not None:
            resolved[i] = nxt

    return np.stack(resolved, axis=0).astype(np.float32)


def _carrier_translation(
    *,
    kp2d: np.ndarray,
    frame_indices: np.ndarray,
    per_frame_K: dict,
    per_frame_R: dict,
    per_frame_t: dict,
    distortion: tuple[float, float],
    root_R: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame ankle-midpoint ray-cast anchor, shared by both
    ``anchor_mode`` strategies in :func:`anchor_root_translation`.

    Ray-casts the ankle-midpoint COCO pixel to the ``_FOOT_PLANE_Z``
    pitch plane every frame and back-solves ``root_t`` so that
    ``offsets[i]`` (the root->foot-midpoint vector, in the root's own
    canonical frame) lands on that ray-cast point once rotated by
    ``root_R[i]``. ``offsets`` is the ONLY thing that differs between
    the two anchor modes: the constant ``_ANKLE_IN_ROOT`` for
    ``ankle_mid``, a per-frame posed-FK offset for ``contact`` — sharing
    this one implementation means the two modes can never numerically
    diverge on the per-frame gating logic itself.

    Frames below ``_ANKLE_CONF_MIN`` confidence, with no camera, or
    whose ray is parallel to the ground plane hold the last successfully
    anchored ``root_t`` (never teleport) and carry an attenuated/zero
    confidence — this is exactly today's (pre-refactor) single-offset
    loop's gating, generalised to accept any per-frame offset.

    Returns ``(root_t, confidence)``. ``root_t`` is NOT smoothed here —
    the caller applies the trailing SavGol (see ``anchor_root_translation``,
    which for ``contact`` mode must smooth the carrier BEFORE the stance
    pin correction is added, not after).
    """
    n = int(frame_indices.shape[0])
    root_t = np.zeros((n, 3), dtype=float)
    confidence = np.zeros(n, dtype=float)
    last_anchored: np.ndarray | None = None
    for i in range(n):
        fi_int = int(frame_indices[i])
        R = per_frame_R.get(fi_int)
        if R is None:
            # No camera for this frame — leave translation/confidence zero.
            continue
        K = per_frame_K[fi_int]
        t = per_frame_t[fi_int]
        kp = kp2d[i]
        left = kp[_COCO_LEFT_ANKLE]
        right = kp[_COCO_RIGHT_ANKLE]
        ankle_conf = float(min(left[2], right[2]))
        if ankle_conf < _ANKLE_CONF_MIN:
            if last_anchored is not None:
                root_t[i] = last_anchored
            confidence[i] = ankle_conf
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
            # Ray parallel to the ground plane — skip this frame.
            if last_anchored is not None:
                root_t[i] = last_anchored
            confidence[i] = 0.0
            continue
        root_t[i] = anchor_translation(foot_world, offsets[i], root_R[i])
        last_anchored = root_t[i]
        confidence[i] = ankle_conf
    return root_t, confidence


def _apply_lean_correction(
    *,
    root_R_pitch: np.ndarray,
    frame_indices: np.ndarray,
    kp2d: np.ndarray,
    per_frame_K: dict,
    per_frame_R: dict,
    per_frame_t: dict,
    distortion: tuple[float, float],
    lean_correction_deg: float,
) -> np.ndarray:
    """Rotate each frame's ``root_R_pitch`` toward the camera by
    ``lean_correction_deg`` — the monocular-HMR lean-away-from-camera
    bias fix. Uses the same ankle-midpoint ray-cast as
    :func:`_carrier_translation` to find the horizontal camera->foot
    direction, but is otherwise independent of ``anchor_mode`` (the
    correction direction doesn't depend on which root->foot offset the
    translation solve will use).

    MUST run BEFORE :func:`anchor_root_translation`, never inside it:
    ``scripts/reanchor_hmr_world.py`` loads a saved ``root_R`` that
    already has this baked in (see that npz's provenance — the stage
    always applies this pre-pass before calling
    ``anchor_root_translation``) and must never re-apply it, which would
    double-lean an already-corrected track.

    Returns a NEW array (never mutates ``root_R_pitch`` in place); frames
    with no confident ray-cast are left unrotated, matching the original
    per-frame gating exactly.
    """
    out = np.array(root_R_pitch, dtype=float, copy=True)
    if lean_correction_deg == 0.0:
        return out
    n = int(frame_indices.shape[0])
    for i in range(n):
        fi_int = int(frame_indices[i])
        R = per_frame_R.get(fi_int)
        if R is None:
            continue
        K = per_frame_K[fi_int]
        t = per_frame_t[fi_int]
        kp = kp2d[i]
        left = kp[_COCO_LEFT_ANKLE]
        right = kp[_COCO_RIGHT_ANKLE]
        ankle_conf = float(min(left[2], right[2]))
        if ankle_conf < _ANKLE_CONF_MIN:
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
            continue
        cam_centre = -R.T @ t
        v_horiz = np.array(
            [foot_world[0] - cam_centre[0],
             foot_world[1] - cam_centre[1],
             0.0],
            dtype=float,
        )
        v_norm = float(np.linalg.norm(v_horiz))
        if v_norm <= 1e-6:
            continue
        v_horiz /= v_norm
        z_up = np.array([0.0, 0.0, 1.0])
        lean_axis = np.cross(v_horiz, z_up)
        lean_axis_norm = float(np.linalg.norm(lean_axis))
        if lean_axis_norm <= 1e-6:
            continue
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
        out[i] = correction_R @ out[i]
    return out


def _savgol_root_t(root_t: np.ndarray, window: int, order: int) -> np.ndarray:
    """Same translation-jitter dampener as the pre-refactor stage: SavGol
    across time on the dense per-frame translation. In ``contact`` mode
    this MUST run on the carrier before any stance-pin delta is added —
    see :func:`anchor_root_translation` — otherwise the pins themselves
    would get smeared by a post-hoc smooth."""
    if window > 1 and root_t.shape[0] >= window:
        return savgol_axis(root_t, window=window, order=order, axis=0)
    return root_t


def anchor_root_translation(
    *,
    kp2d: np.ndarray,
    frame_indices: np.ndarray,
    per_frame_K: dict,
    per_frame_R: dict,
    per_frame_t: dict,
    distortion: tuple[float, float],
    thetas: np.ndarray,
    root_R: np.ndarray,
    betas: np.ndarray,
    cfg: dict,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, FootContacts | None]:
    """Per-frame root translation anchoring — shared by the ``hmr_world``
    stage and ``scripts/reanchor_hmr_world.py`` (plan Task 5, spec
    §2[C]). ``root_R`` must ALREADY carry any lean correction (see
    :func:`_apply_lean_correction`) — this function never mutates
    rotation, only reads it.

    ``cfg["anchor_mode"]`` selects the strategy:

    - ``"ankle_mid"`` (legacy): today's canonical-offset ankle-midpoint
      ray-cast anchor, reproduced with BIT-PARITY (the constant
      ``_ANKLE_IN_ROOT``, hold-last low-confidence handling, trailing
      SavGol). Returns ``contacts=None`` — there is no per-foot contact
      signal in this mode.
    - ``"contact"`` (default): the SAME ray-cast, but the offset is the
      per-frame POSED-FK mid-ankle position
      (``0.5*(canon[:,7]+canon[:,8])``, canonical-frame lateral zeroed,
      rotated by ``root_R``) instead of the canonical constant — this
      dense carrier is SavGol-smoothed (BEFORE any per-foot pin
      correction, so pins are never smeared by a post-hoc smooth), then
      :func:`src.utils.foot_contact.detect_contacts` finds per-foot
      stance spans and :func:`src.utils.foot_lock.solve_root_with_pins`
      pins the root so the stance foot stops sliding.

    ``cfg`` mirrors ``config/default.yaml``'s ``hmr_world`` section:
    ``anchor_mode``, ``root_t_savgol_window``/``root_t_savgol_order``,
    and (contact mode only) the ``contact`` sub-dict consumed by
    ``detect_contacts``/``solve_root_with_pins``.

    Confidence is ``min(left_ankle_conf, right_ankle_conf)`` per frame
    (0 when the camera is absent or the ray is parallel to the ground) —
    NOT additionally floored by GVHMR's per-joint confidence the way the
    pre-refactor code was: that signal is transient (never persisted to
    the ``*_smpl_world.npz``), so a reanchor running from saved sidecars
    alone cannot reconstruct it. The floor rarely bound in practice
    (per-joint confidence is usually well above the ankle threshold).
    """
    frame_indices = np.asarray(frame_indices)
    root_R = np.asarray(root_R, dtype=float)
    n = int(frame_indices.shape[0])
    anchor_mode = str(cfg.get("anchor_mode", "contact"))
    savgol_window = int(cfg.get("root_t_savgol_window", 5))
    savgol_order = int(cfg.get("root_t_savgol_order", 2))

    if anchor_mode == "ankle_mid":
        offsets = np.tile(_ANKLE_IN_ROOT, (n, 1))
        root_t, confidence = _carrier_translation(
            kp2d=kp2d, frame_indices=frame_indices,
            per_frame_K=per_frame_K, per_frame_R=per_frame_R,
            per_frame_t=per_frame_t, distortion=distortion,
            root_R=root_R, offsets=offsets,
        )
        root_t = _savgol_root_t(root_t, savgol_window, savgol_order)
        return root_t, confidence, None

    # "contact" mode: posed-FK mid-ankle carrier + stance-pinned solve.
    rest_joints = beta_adjusted_rest_joints(betas, load_smpl_neutral_model())
    canon = compute_canonical_joints_batch(np.asarray(thetas), rest_joints)
    offsets = 0.5 * (canon[:, 7, :] + canon[:, 8, :])
    offsets[:, 0] = 0.0  # zero lateral, matches _ANKLE_IN_ROOT's convention

    root_t_carrier, confidence = _carrier_translation(
        kp2d=kp2d, frame_indices=frame_indices,
        per_frame_K=per_frame_K, per_frame_R=per_frame_R,
        per_frame_t=per_frame_t, distortion=distortion,
        root_R=root_R, offsets=offsets,
    )
    root_t_carrier = _savgol_root_t(root_t_carrier, savgol_window, savgol_order)

    contact_cfg = cfg.get("contact") or {}
    contacts = detect_contacts(
        kp2d=kp2d, frame_indices=frame_indices,
        per_frame_K=per_frame_K, per_frame_R=per_frame_R,
        per_frame_t=per_frame_t, distortion=distortion,
        thetas=thetas, root_R=root_R, betas=betas, fps=fps, cfg=contact_cfg,
    )
    root_t, _stats = solve_root_with_pins(
        root_carrier=root_t_carrier, root_R=root_R, thetas=thetas, betas=betas,
        contacts=contacts, fps=fps,
        max_correction_m=float(contact_cfg.get("max_correction_m", 0.5)),
        decay_s=float(contact_cfg.get("decay_s", 0.6)),
        rest_joints=rest_joints,
    )
    return root_t, confidence, contacts


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
    fps: float = 25.0,
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

    # Dense per-track-frame calibrated camera R (world-to-camera), for
    # GVHMR's camera-rotation conditioning. When available this replaces
    # GVHMR's internal SimpleVO estimate — see build_track_camera_R.
    gvhmr_R = build_track_camera_R(track_frames, per_frame_R)

    hmr_out = run_on_track(
        track_frames=track_frames,
        video_path=video_path,
        checkpoint=Path(cfg.get("checkpoint", "")),
        device=str(cfg.get("device", "auto")),
        extractor_device=str(cfg.get("extractor_device", "cpu")),
        batch_size=int(cfg.get("batch_size", 16)),
        max_sequence_length=int(cfg.get("max_sequence_length", 120)),
        estimator=estimator,
        per_frame_K=gvhmr_K,
        per_frame_R=gvhmr_R,
    )
    thetas = np.asarray(hmr_out["thetas"])             # (N, 24, 3)
    betas_all = np.asarray(hmr_out["betas"])           # (N, 10)
    root_R_cam = np.asarray(hmr_out["root_R_cam"])     # (N, 3, 3)
    # hmr_out["joint_confidence"] (per-joint GVHMR confidence) used to
    # floor the anchored-translation confidence below; that floor moved
    # out when the translation solve was factored into
    # anchor_root_translation (see that function's docstring) because
    # the reanchor script — which shares the same code — has no access
    # to this transient, never-persisted signal.
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

    # 3. Convert root rotation to pitch frame and SLERP-smooth. Frames
    # with no camera entry get an identity root_R placeholder — the
    # translation solve below (anchor_root_translation) independently
    # gates on per_frame_R and leaves those frames at zero confidence.
    frame_indices = np.array([fi for fi, _ in track_frames])
    root_R_pitch = np.empty_like(root_R_cam)
    for i, fi in enumerate(frame_indices):
        R_t = per_frame_R.get(int(fi))
        if R_t is None:
            root_R_pitch[i] = np.eye(3)
            continue
        root_R_pitch[i] = smpl_root_in_pitch_frame(root_R_cam[i], R_t)
    root_R_pitch = slerp_window(root_R_pitch, window=slerp_w)

    # 4. θ Savgol smoothing across time, per-joint, per-axis.
    thetas_smooth = savgol_axis(
        thetas, window=savgol_window, order=savgol_order, axis=0
    ).astype(np.float32)

    # 5. Lean-correction pre-pass — the same ankle-mid ray-cast the
    # translation solve below uses, applied to root_R BEFORE anchoring
    # (never inside anchor_root_translation — see that function's
    # docstring for why: the reanchor script consumes a saved root_R
    # that already has this baked in and must not re-apply it).
    root_R_pitch = _apply_lean_correction(
        root_R_pitch=root_R_pitch,
        frame_indices=frame_indices,
        kp2d=kp2d,
        per_frame_K=per_frame_K,
        per_frame_R=per_frame_R,
        per_frame_t=per_frame_t,
        distortion=distortion,
        lean_correction_deg=lean_correction_deg,
    )

    # 6. Contact-aware (or legacy ankle-mid) root translation anchoring —
    # shared with scripts/reanchor_hmr_world.py via anchor_root_translation
    # so a local re-solve from saved sidecars is bit-identical to what a
    # fresh stage run would compute. Frames with no camera never get a
    # per_frame_R entry, so no separate gate is needed here
    # (anchor_root_translation checks per_frame_R itself).
    anchor_mode = str(cfg.get("anchor_mode", "contact"))
    anchor_cfg = dict(cfg)
    anchor_cfg["root_t_savgol_window"] = root_t_savgol_window
    anchor_cfg["root_t_savgol_order"] = root_t_savgol_order
    root_t, confidence, contacts = anchor_root_translation(
        kp2d=kp2d,
        frame_indices=frame_indices,
        per_frame_K=per_frame_K,
        per_frame_R=per_frame_R,
        per_frame_t=per_frame_t,
        distortion=distortion,
        thetas=thetas_smooth,
        root_R=root_R_pitch,
        betas=betas,
        cfg=anchor_cfg,
        fps=fps,
    )

    # 7. (No ground snap.) The previous ``ground_snap_z`` post-process
    # halved root_t.z every frame whose per-frame velocity was below
    # threshold — which is every frame for a stationary or slowly-
    # moving player. That collapsed the pelvis toward z=0 (so the
    # avatar's feet ended up below the pitch). The foot-anchor
    # ray-cast above already constrains the pelvis position
    # consistently with the ankle keypoint, so no extra snap is
    # needed. ``ground_snap_velocity`` is kept in the signature for
    # backwards-compat but is now ignored.
    _ = ground_snap_velocity

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

    # Contact-mode sidecar: per-foot stance spans + pins so refined_poses
    # (and the dashboard/eval harness) don't have to re-derive them.
    # ankle_mid mode has no contact signal (contacts is None) — nothing
    # to write.
    if contacts is not None:
        save_foot_contacts(
            out_dir / f"{out_key}_foot_contacts.json", contacts,
            shot_id=shot_id, player_id=player_id, anchor_mode=anchor_mode,
        )

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
